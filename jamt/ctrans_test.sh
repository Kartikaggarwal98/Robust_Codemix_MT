RAW_DATA=examples/translation/ctrans/
SAVE_DATA=data-bin/ctrans

src_lang=hicmrom
tgt_lang=en

MODEL_PATH=checkpoints/ctrans_rcmt1
echo "Testing model for RCMT1...."
fairseq-generate \
    $SAVE_DATA \
    --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,en-hicmrom \
    -s $src_lang -t $tgt_lang \
    --path $MODEL_PATH/checkpoint_best.pt \
    --beam 5 --remove-bpe=sentencepiece > preds.txt

# MODEL_PATH=checkpoints/ctrans_rcmt2
# echo "Testing model for RCMT2...."
# fairseq-generate \
#     $SAVE_DATA \
#     --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,hicm-en,en-hicmrom,en-hicm \
#     -s $src_lang -t $tgt_lang \
#     --path $MODEL_PATH/checkpoint_best.pt \
#     --beam 5 --remove-bpe=sentencepiece > preds.txt

# MODEL_PATH=checkpoints/ctrans_zcmt
# echo "Testing model for ZCMT...."
# fairseq-generate \
#     $SAVE_DATA \
#     --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,en-hicmrom,hicm-en,en-hicm,bn-en,bnrom-en,en-bn,en-bnrom \
#     -s $src_lang -t $tgt_lang \
#     --path $MODEL_PATH/checkpoint_best.pt \
#     --beam 5 --remove-bpe=sentencepiece > preds.txt


### if we want to use any random test set, first encode that and then use fairseq-interactive for line-by-line translation
# SPM_ENCODE=scripts/spm_encode.py
# echo "encoding test with learned BPE..."
# python "$SPM_ENCODE" \
#     --model "$RAW_DATA/sentencepiece.uni.model" \
#     --output_format=piece \
#     --inputs test.txt \
#     --outputs test_sp.txt

# cat test_sp.txt \
# | fairseq-interactive $SAVE_DATA \
#     --task multilingual_translation --lang-pairs noisyhicmrom-en,hicmrom-en,hicm-en,en-hicmrom,en-hicm \
#     -s $src_lang -t $tgt_lang --buffer-size 1000 \
#     --path $MODEL_PATH/checkpoint_best.pt \
#     --beam 5 --remove-bpe=sentencepiece \
#     > preds_.txt

# grep ^H preds_.txt | cut -f3 > preds2.txt
# fairseq-score --sys preds2.txt --ref test.en.txt

