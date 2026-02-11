import sys
import os
import time
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'grover'))
from grover.model.models import GROVEREmbedding
from grover.util.scheduler import NoamLR
from grover.util.utils import create_logger
from grover.data import MolCollator
from task.train import load_finetune_data
from fusion_model import GroverFinetuneTask

finetune_args = Namespace(
    bond_drop_rate=0.2,
    parser_name='finetune', no_cache=True, gpu=0, batch_size=32, tensorboard=False,
    data_path='grover/exampledata/finetune/ssl_molecules_prot.csv', # update path
    use_compound_names=False, max_data_size=None, features_only=False, features_generator=None,
    features_path=['grover/exampledata/finetune/ssl_molecules.npz'], # update path
    save_dir='grover/model/finetune/ssl_molecules/', # update path
    save_smiles_splits=False, checkpoint_dir=None,
    checkpoint_path=None, dataset_type='regression', separate_val_path=None, separate_val_features_path=None,
    separate_test_path=None, separate_test_features_path=None, split_type='scaffold_balanced', split_sizes=[0.8, 0.1, 0.1], num_folds=1,
    folds_file=None, val_fold_index=None, test_fold_index=None, crossval_index_dir=None, crossval_index_file=None, seed=0, metric='rmse',
    show_individual_scores=False, epochs=10, warmup_epochs=2.0, init_lr=0.00015, max_lr=0.001, final_lr=0.0001, early_stop_epoch=1000,
    ensemble_size=1, dropout=0.0, activation='ReLU', ffn_hidden_size=200, ffn_num_layers=2, weight_decay=0.0, select_by_loss=False,
    embedding_output_type='atom', self_attention=False, attn_hidden=4, attn_out=128, dist_coff=0.1, distinct_init=False,
    fine_tune_coff=1, enbl_multi_gpu=False, cuda=True, features_scaling=False, minimize_score=False, checkpoint_paths=None,
    use_input_features=['grover/exampledata/finetune/ssl_molecules.npz'], # Updated path
    num_lrs=1, fingerprint=False, cross_attn_hsize=128
)

model_args = Namespace(
    cuda=True, epochs=200,
    weight_decay=0.0, warmup_epochs=2.0, init_lr=0.00015, max_lr=0.001, final_lr=0.0001, fine_tune_coff=1,
    es_patience_rounds=2, es_min_delta=1e-4, val_every_nepoch=2,
    # fusion model sizes
    prot_vocab_size = 21, prot_emb_dim=128, prot_hidden_size=128, grover_hidden_size=128,
    attn_nheads=4,
    # final MLP head
    ffn_hidden_size=200, ffn_num_layers=2, dropout=0.1, activation='PReLU',
    save_dir='grover/model/finetune/ssl_molecules/' # update path
)

def main():
    logger = create_logger(name='train', save_dir=finetune_args.save_dir, quiet=False)
    debug, info = logger.debug, logger.info

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_finetune_data(finetune_args, debug, logger)

    pretrained_path = 'grover/model/ssl/model.ep500' # load path
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained model not found at {pretrained_path}")
        
    state = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only=False)
    pretrain_args, loaded_state_dict = state['args'], state['state_dict']
    emb = GROVEREmbedding(pretrain_args)

    model = GroverFinetuneTask(emb, model_args)
    optimizer = optim.Adam(model.parameters(), lr=model_args.init_lr, weight_decay=model_args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss     = float('inf')
    best_val_epoch    = 0
    no_improve_rounds = 0

    mol_collator = MolCollator(args=finetune_args, shared_dict=shared_dict)
    train_loader = DataLoader(train_data, batch_size=finetune_args.batch_size, shuffle=True, collate_fn=mol_collator)
    val_loader = DataLoader(val_data, batch_size=finetune_args.batch_size, shuffle=False, collate_fn=mol_collator)

    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=model_args.warmup_epochs,
        total_epochs=model_args.epochs,
        steps_per_epoch=len(train_loader),
        init_lr=model_args.init_lr,
        max_lr=model_args.max_lr,
        final_lr=model_args.final_lr,
        fine_tune_coff=model_args.fine_tune_coff,
        fine_tune_param_idx=0
    )

    # training loop
    device = torch.device('cuda') if model_args.cuda else torch.device('cpu')
    model.to(device)

    for epoch in range(1, model_args.epochs+1):
        epoch_start = time.time()
        
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False)
        for smiles, graph_comp, feats, mask, targets, prot_idxs, prot_lens in loop:
            graph_comp = tuple(
                comp.to(device) if isinstance(comp, torch.Tensor) else comp
                for comp in graph_comp
            )
            mask      = mask.to(device)
            targets   = targets.to(device)
            prot_idxs = [p.to(device) for p in prot_idxs]
            
            optimizer.zero_grad()
            preds = model(smiles, graph_comp, feats, mask, targets, prot_idxs, prot_lens)
            loss = criterion(preds.view(-1), targets.view(-1)).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            loop.set_postfix(train_loss=loss.item())
        
        current_lr = scheduler.get_lr()[0]
        avg_train_loss = sum(train_losses) / len(train_losses)
        epoch_time   = time.time() - epoch_start
        debug(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} learning_rate={current_lr:.6f} completed in {epoch_time:.0f}s")
        
        # validation
        if epoch % model_args.val_every_nepoch == 0:
            checkpoint = {
                'training_args': model_args,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val': best_val_loss,
                'no_improve_rounds': no_improve_rounds
            }
            
            os.makedirs(model_args.save_dir, exist_ok=True)
            torch.save(checkpoint, f"{model_args.save_dir}/checkpoint.pt") # save
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"[Epoch {epoch}] Validating", leave=False)
                for smiles, graph_comp, feats, mask, targets, prot_idxs, prot_lens in loop:
                    graph_comp = tuple(
                        comp.to(device) if isinstance(comp, torch.Tensor) else comp
                        for comp in graph_comp
                    )
                    mask      = mask.to(device)
                    targets   = targets.to(device)
                    prot_idxs = [p.to(device) for p in prot_idxs]
        
                    preds = model(smiles, graph_comp, feats, mask, targets, prot_idxs, prot_lens)
                    loss = criterion(preds.view(-1), targets.view(-1)).mean()
                    val_losses.append(loss.item())
                    loop.set_postfix(val_loss=loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            if avg_val_loss <= best_val_loss * (1 - model_args.es_min_delta):
                best_val_loss     = avg_val_loss
                no_improve_rounds = 0
                best_val_epoch    = epoch
            else:
                no_improve_rounds += 1
            debug(f"Epoch {epoch} val_loss={avg_val_loss:.4f} rounds={no_improve_rounds:d}")
            
            if no_improve_rounds > model_args.es_patience_rounds:
                info(f"No improvement after {no_improve_rounds} validation rounds "
                     f"(last best epoch {best_val_epoch}), stopping early.")
                break

    torch.save(model.state_dict(), f"{model_args.save_dir}/model.pt")

if __name__ == "__main__":
    main()