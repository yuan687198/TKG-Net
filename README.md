# GPT-based Knowledge Guiding Network for Commonsense Video Caption
There will be more updates in the future.

## Background
This repository is for the new task of [video-based commonsense captioning](https://arxiv.org/abs/2003.05162), which aims to generate event-wise captions and meanwhile provide multiple commonsense descriptions (e.g., attribute, effect and intention) about the underlying event in the video.

## Dataset
V2C dataset in [V2C_annotations.zip](https://pan.baidu.com/s/1Ayq6Y4lnU8x5eEdrU6GFtQ) (Extraction code: gr8f), which consists:
    
    V2C_annotations.zip
    ├── msrvtt_new_info.json                      # MSR-VTT captions and token dictionary.
    ├── v2c_info.json                             # V2C Raw, captions/CMS, and token dictionary.
    ├── V2C_MSR-VTT_caption.json                  # V2C Raw, captions/CMS after tokenization.
    ├── train_cvpr_humanRank_V2C_caption.json     # a human re-verified clean split for V2C annotations.
    └── v2cqa_v1_train.json                       # for V2C QA, consisting of captions, CMS, and CMS related questions/answers.

## Video Features
We use the pre-trained models including ResNet152, SoundNet and I3D  to extract the appearance feature, audio feature and motion feature, respectively. Video Features data can be obtained in the [link](https://pan.baidu.com/s/1oDLOPuuXuLc3TAFrOzn3Rg) (Extraction code: d0ds).


## Training and Evaluation
* Environment: This implementation was complimented on PyTorch-1.8.1.
                
You first need to complete the training of the first-stage gpt.

E.g., to initiate a training on **intention** prediction tasks (set --cms 'int'), with 1 RNN video encoder layer, and 6 transformer decoder layers with 8 attention heads, 64 head dim, and 1024 model dim, for 1000 epochs under CUDA mode, and shows intermedia generation examples:
```python
python train_bert.py --cms int --batch_size 128 --epochs 1000 --model_name /data/yuanmq/hybridnet/0.2all_lr3e-4.pt --num_layer 6 --dim_head 64 --dim_inner 1024  --num_head 8 --dim_vis_feat 2048 --dropout 0.1 --rnn_layer 1 --checkpoint_path ./save/intentionbert2000 --info_json data/v2c_info.json --caption_json data/V2C_MSR-VTT_caption.json  --print_loss_every 10 --cuda --show_predict   
```
You can download the 0.2all_lr3e-4.pt model in the [link](https://pan.baidu.com/s/1iIukOY8_FiqNgiOih_GoWQ) (Extraction code: 86mv).

For completion evaluations:
```python
python test_cms.py  --cms int --batch_size 64 --num_layer 6 --dim_head 64 --dim_inner 1024 \
                    --num_head 8 --dim_vis_feat 2048 --dropout 0.1 --rnn_layer 1 --checkpoint_path ./save  \
                    --info_json data/v2c_info.json --caption_json data/V2C_MSR-VTT_caption.json  \
                    --load_checkpoint save/**.pth --cuda
```
For generation evaluations:
```python
python test_cap2cms.py  --cms int --batch_size 64 --num_layer 6 --dim_head 64 --dim_inner 1024 \
                        --num_head 8 --dim_vis_feat 2048 --dropout 0.1 --rnn_layer 1 --checkpoint_path ./save  \
                        --info_json data/v2c_info.json --caption_json data/V2C_MSR-VTT_caption.json \
                        --load_checkpoint save/*.pth --cuda
```


