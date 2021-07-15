RAW_DATA=examples/translation/calign/
SAVE_DATA=data-bin/calign


#### rom: romanized, cm: codemix, hi: hindi, bn: bengali

### For training rcmt1 model (3 language directions)
echo "Training model for RCMT1...."
MODEL_PATH=checkpoints/calign_rcmt1
mkdir -p $MODEL_PATH

###### if cuda is available add devices accordingly
fairseq-train $SAVE_DATA \
    --max-epoch 50 \
    --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,en-hicmrom \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-encoders --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir $MODEL_PATH \
    --max-tokens 4000 \
    --update-freq 8 


### For training rcmt2 model (5 language directions)
# echo "Training model for RCMT2...."
# MODEL_PATH=checkpoints/calign_rcmt2
# mkdir -p $MODEL_PATH

# fairseq-train $SAVE_DATA \
#     --max-epoch 50 \
#     --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,hicm-en,en-hicmrom,en-hicm \
#     --arch multilingual_transformer_iwslt_de_en \
#     --share-decoders --share-encoders --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 --warmup-init-lr '1e-07' \
#     --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --save-dir $MODEL_PATH \
#     --max-tokens 4000 \
#     --update-freq 8 


### For training zcmt model (9 language directions)
# echo "Training model for ZCMT...."
# MODEL_PATH=checkpoints/calign_zcmt
# mkdir -p $MODEL_PATH

# CUDA_VISIBLE_DEVICES=0 fairseq-train $SAVE_DATA \
#     --max-epoch 50 \
#     --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,en-hicmrom,hicm-en,en-hicm,bn-en,bnrom-en,en-bn,en-bnrom \
#     --arch multilingual_transformer_iwslt_de_en \
#     --share-decoders --share-encoders --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9, 0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 --warmup-init-lr '1e-07' \
#     --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --save-dir $MODEL_PATH \
#     --max-tokens 4000 \
#     --update-freq 8 --fp16

