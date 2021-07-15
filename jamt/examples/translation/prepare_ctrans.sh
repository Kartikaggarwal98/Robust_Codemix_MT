

ROOT=$(dirname "$0")
SCRIPTS=$ROOT/../../scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

# BPESIZE=32000
BPESIZE=1600
DATA=$ROOT/ctrans/
mkdir -p "$DATA"

TRAIN_MINLEN=1  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens
echo $ROOT $DATA

SRCS=(
    "hicm"
    "hicmrom"
    "noisyhicmrom"
    "bn"
    "bnrom"
)
TGT=en

##learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.ctrans.${SRC}-en.${SRC}; done | tr "\n" ",")$DATA/train.ctrans.hicm-en.en
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.uni \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=unigram \
    --input_sentence_size=4000000 --shuffle_input_sentence=true

# #### encode train/valid
echo "encoding train with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATA/sentencepiece.uni.model" \
    --output_format=piece \
    --inputs $DATA/train.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/train.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/train.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/train.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/train.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/train.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/train.ctrans.${SRCS[3]}-$TGT.${TGT} \
    --outputs $DATA/train.uni.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/train.uni.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/train.uni.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/train.uni.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/train.uni.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/train.uni.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/train.uni.ctrans.${SRCS[3]}-$TGT.${TGT} \
    --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN

echo "encoding valid with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATA/sentencepiece.uni.model" \
    --output_format=piece \
    --inputs $DATA/valid.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/valid.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/valid.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/valid.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/valid.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/valid.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/valid.ctrans.${SRCS[3]}-$TGT.${TGT} \
    --outputs $DATA/valid.uni.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/valid.uni.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/valid.uni.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/valid.uni.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/valid.uni.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/valid.uni.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/valid.uni.ctrans.${SRCS[3]}-$TGT.${TGT} 

echo "encoding test with learned BPE..."
python "$SPM_ENCODE" \
    --model "$DATA/sentencepiece.uni.model" \
    --output_format=piece \
    --inputs $DATA/test.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/test.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/test.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/test.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/test.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/test.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/test.ctrans.${SRCS[3]}-$TGT.${TGT} \
    --outputs $DATA/test.uni.ctrans.${SRCS[0]}-$TGT.${SRCS[0]} $DATA/test.uni.ctrans.${SRCS[1]}-$TGT.${SRCS[1]} $DATA/test.uni.ctrans.${SRCS[2]}-$TGT.${SRCS[2]} \
    $DATA/test.uni.ctrans.${SRCS[3]}-$TGT.${SRCS[3]} $DATA/test.uni.ctrans.${SRCS[4]}-$TGT.${SRCS[4]} \
    $DATA/test.uni.ctrans.${SRCS[0]}-$TGT.${TGT} $DATA/test.uni.ctrans.${SRCS[3]}-$TGT.${TGT} 

# ## if joint vocab in fairseq, add dummy count to uni.vocab
cd $DATA
tail -n +4 "sentencepiece.uni.vocab" | cut -f1 | sed 's/$/ 100/g' > "fairseq.vocab"

cp train.uni.ctrans.hicm-en.en train.uni.ctrans.hicmrom-en.en
cp train.uni.ctrans.hicm-en.en train.uni.ctrans.noisyhicmrom-en.en

cp valid.uni.ctrans.hicm-en.en valid.uni.ctrans.hicmrom-en.en
cp valid.uni.ctrans.hicm-en.en valid.uni.ctrans.noisyhicmrom-en.en

cp test.uni.ctrans.hicm-en.en test.uni.ctrans.hicmrom-en.en
cp test.uni.ctrans.hicm-en.en test.uni.ctrans.noisyhicmrom-en.en

cp train.uni.ctrans.bn-en.en train.uni.ctrans.bnrom-en.en
cp valid.uni.ctrans.bn-en.en valid.uni.ctrans.bnrom-en.en
cp test.uni.ctrans.bn-en.en test.uni.ctrans.bnrom-en.en
