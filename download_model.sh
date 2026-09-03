#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf"
PRO_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
# 0731 refresh (2026-08-01): the checkpoint the fork's launch defaults prefer.
Q2_0731_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
Q2_Q4_0731_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
Q4_0731_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
MXFP4_0731_FILE="DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf"
PRO_0813_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf"
# Vision-Exp (2026-08-31): a separate continued-training checkpoint; this
# fork serves its TEXT side (opt-in) -- image input is not implemented here.
VISION_Q2_FILE="DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
VISION_Q2_Q4_FILE="DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
VISION_MXFP4_FILE="DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf"
VISION_ENCODER_FILE="DeepSeek-V4-Flash-Vision-Encoder.gguf"
# The fork's DSpark drafters (extracted with gguf-tools/dspark_extract.py,
# the 0731 recipe) are hosted separately from the language GGUFs.
DRAFTER_REPO="${DSPARK_HF_REPO:-bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF}"
DRAFTER_0731_FILE="DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf"
DRAFTER_VISION_FILE="DSpark-drafter-Q2K-Q8-MarkovQ8-vision-exp.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 GGUF downloader

Usage:
  ./download_model.sh q2-0731 [--token TOKEN]
  ./download_model.sh q2-q4-0731 [--token TOKEN]
  ./download_model.sh q4-0731 [--token TOKEN]
  ./download_model.sh mxfp4-0731 [--token TOKEN]
  ./download_model.sh pro-0813 [--token TOKEN]
  ./download_model.sh vision-q2 [--token TOKEN]
  ./download_model.sh vision-q2-q4 [--token TOKEN]
  ./download_model.sh vision-mxfp4 [--token TOKEN]
  ./download_model.sh vision-encoder [--token TOKEN]
  ./download_model.sh drafter-0731 [--token TOKEN]
  ./download_model.sh drafter-vision [--token TOKEN]
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh q2 [--token TOKEN]
  ./download_model.sh q4 [--token TOKEN]
  ./download_model.sh pro [--token TOKEN]
  ./download_model.sh pro-imatrix [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]

Targets:
  *** CURRENT CHECKPOINTS (the -0731 refresh is what ds4-server picks by default) ***

  q2-0731
       DeepSeek V4 Flash 0731, 2-bit routed experts (imatrix), about 81 GB.
       Recommended for 96 and 128 GB machines; auto-selected by ds4-server.

  q2-q4-0731
       0731 mixed quant: q2 routed experts with layers 37-42 at q4, about 98 GB.

  q4-0731
       0731 4-bit routed experts (imatrix), about 153 GB. 256 GB RAM or more.

  mxfp4-0731
       0731 native MXFP4 routed experts, about 156 GB.

  pro-0813
       DeepSeek V4 PRO 0813 q2 imatrix quant, about 430 GB; 512 GB machines.

  vision-q2 / vision-q2-q4 / vision-mxfp4
       DeepSeek V4 Flash Vision-Exp (2026-08-31), a separate continued-training
       checkpoint; the matching vision encoder (0.9 GB) is downloaded alongside.
       This fork serves its TEXT side: pass -m <file> explicitly (opt-in), and
       pair it only with the Vision-Exp DSpark drafter -- the engine refuses a
       0731 drafter beside it. Image input is not implemented in this fork yet.

  vision-encoder
       The standalone 0.9 GB Vision-Exp encoder, when the language GGUF is
       already present.

  drafter-0731 / drafter-vision
       The fork's DSpark speculative-decoding drafter (6.46 GiB) for the
       matching checkpoint, from $DRAFTER_REPO. ds4-server auto-attaches the
       drafter that sits beside its base GGUF; generations never mix.
       Since 2026-09-03 both carry a Q8_0 Markov table ("MarkovQ8" in the
       name); the earlier F16-Markov files keep loading and are used when
       the MarkovQ8 file is absent.

  *** PREVIOUS GENERATION (pre-0731): PREFER THE IMATRIX VERSIONS BELOW ***

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-imatrix
       DeepSeek V4 PRO imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  Legacy GGUF files:

  q2   2-bit routed experts, about 81 GB on disk.
       Older non-imatrix model for 96 and 128 GB RAM machines. Prefer
       q2-imatrix unless you specifically need the legacy quant.

  q4   4-bit routed experts, about 153 GB on disk.
       Older non-imatrix model for machines with 256 GB RAM or more. Prefer
       q4-imatrix unless you specifically need the legacy quant.

  pro  DeepSeek V4 PRO non-imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines. Prefer pro-imatrix unless you
       specifically need the legacy quant.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q4-imatrix, q2, and q4, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift

MODEL_FILES=
case "$MODEL" in
    q2-0731) MODEL_FILE=$Q2_0731_FILE ;;
    q2-q4-0731) MODEL_FILE=$Q2_Q4_0731_FILE ;;
    q4-0731) MODEL_FILE=$Q4_0731_FILE ;;
    mxfp4-0731) MODEL_FILE=$MXFP4_0731_FILE ;;
    pro-0813) MODEL_FILE=$PRO_0813_FILE ;;
    vision-q2) MODEL_FILE=$VISION_Q2_FILE; MODEL_FILES="$MODEL_FILE $VISION_ENCODER_FILE" ;;
    vision-q2-q4) MODEL_FILE=$VISION_Q2_Q4_FILE; MODEL_FILES="$MODEL_FILE $VISION_ENCODER_FILE" ;;
    vision-mxfp4) MODEL_FILE=$VISION_MXFP4_FILE; MODEL_FILES="$MODEL_FILE $VISION_ENCODER_FILE" ;;
    vision-encoder) MODEL_FILE=$VISION_ENCODER_FILE ;;
    drafter-0731) MODEL_FILE=$DRAFTER_0731_FILE; FILE_REPO=$DRAFTER_REPO ;;
    drafter-vision) MODEL_FILE=$DRAFTER_VISION_FILE; FILE_REPO=$DRAFTER_REPO ;;
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    q2) MODEL_FILE=$Q2_FILE ;;
    q4) MODEL_FILE=$Q4_FILE ;;
    pro) MODEL_FILE=$PRO_FILE ;;
    pro-imatrix) MODEL_FILE=$PRO_IMATRIX_FILE ;;
    mtp) MODEL_FILE=$MTP_FILE ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    file=$1
    repo=${2:-$REPO}
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$repo/resolve/main/$file"

    mkdir -p "$OUT_DIR"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$repo"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

for f in ${MODEL_FILES:-$MODEL_FILE}; do
    download_one "$f" "${FILE_REPO:-$REPO}"
done

if [ "$MODEL" = "drafter-0731" ] || [ "$MODEL" = "drafter-vision" ]; then
    echo
    echo "Drafter downloaded beside your GGUFs; ds4-server auto-attaches it when the"
    echo "matching base sits in the same directory (or pass --dspark $OUT_DIR/$MODEL_FILE)."
elif [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q4-imatrix, q2, and q4."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" = "vision-encoder" ]; then
    echo
    echo "Vision encoder downloaded. This fork does not consume it yet (image input"
    echo "is not implemented); it is kept beside the Vision-Exp language GGUF for"
    echo "when it is."
else
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
    case "$MODEL" in
        vision-*)
            echo
            echo "Vision-Exp is opt-in in this fork: ./ds4flash.gguf now points at it, but a"
            echo "bare ds4-server prefers the 0731 GGUF in ~/gguf, so start it explicitly:"
            echo "  ./ds4-server -m ./ds4flash.gguf -c 65536"
            echo "ds4-server auto-attaches only its own drafter (./download_model.sh"
            echo "drafter-vision); the engine refuses the 0731 drafter beside it. Text"
            echo "serving only (image/file content blocks are refused with a 400)."
            ;;
    esac
fi

echo
echo "Done."
