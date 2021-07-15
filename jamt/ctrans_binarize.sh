RAW_DATA=examples/translation/ctrans/
SAVE_DATA=data-bin/ctrans

mkdir $SAVE_DATA
cp $RAW_DATA/fairseq.vocab $SAVE_DATA

LANGS=(
    "noisyhicmrom-en"
    "hicm-en"
    "hicmrom-en"
    "bn-en"
    "bnrom-en"
)

for ((i=0;i<${#LANGS[@]};++i)); do
    LANG=${LANGS[i]}
    SRC="$(cut -d'-' -f1 <<<$LANG)"
    TGT="$(cut -d'-' -f2 <<<$LANG)"
    echo "Preparing data for $LANG......"
    fairseq-preprocess --source-lang $SRC --target-lang $TGT \
                   --trainpref $RAW_DATA/train.uni.ctrans.$SRC-$TGT \
                   --validpref $RAW_DATA/valid.uni.ctrans.$SRC-$TGT \
                   --testpref $RAW_DATA/test.uni.ctrans.$SRC-$TGT \
                   --destdir $SAVE_DATA/ \
                   --srcdict $SAVE_DATA/fairseq.vocab \
                   --tgtdict $SAVE_DATA/fairseq.vocab \
                   --workers 40 --fp16
done

LANGS=(
    "en-bn"
    "en-bnrom"
    "en-hicmrom"
    "en-hicm"
)

for ((i=0;i<${#LANGS[@]};++i)); do
    LANG=${LANGS[i]}
    SRC="$(cut -d'-' -f1 <<<$LANG)"
    TGT="$(cut -d'-' -f2 <<<$LANG)"
    echo "Preparing data for $LANG......"
    fairseq-preprocess --source-lang $SRC --target-lang $TGT \
                   --trainpref $RAW_DATA/train.uni.ctrans.$TGT-$SRC \
                   --validpref $RAW_DATA/valid.uni.ctrans.$TGT-$SRC \
                   --testpref $RAW_DATA/test.uni.ctrans.$TGT-$SRC \
                   --destdir $SAVE_DATA/ \
                   --srcdict $SAVE_DATA/fairseq.vocab \
                   --tgtdict $SAVE_DATA/fairseq.vocab \
                   --workers 40 --fp16
done
