#!/bin/bash

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

datasets=("chatlogs" "wikitexts" "state_of_the_union")
chunking_strategies=("fixed_size" "recursive")
embedding_models=("all-MiniLM-L6-v2" "multi-qa-mpnet-base-dot-v1" "text-embedding-3-small" "text-embedding-3-large")
top_k_values=(1 3 5 10)
enable_reranker_set=("yes" "no")

function get_user_choice() {
    PS3=$'\e[33m'"$1"$'\e[0m'
    a=$2[@]
    names=("${!a}")
    selected=()
    select name in "${names[@]}" ; do
        for reply in $REPLY ; do
            selected+=(${names[reply - 1]})
        done
        [[ $selected ]] && break
    done
    echo "${selected[@]}"
}

echo -e "${BLUE}${BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
echo -e "${BLUE}${BOLD}┃     ${MAGENTA}RAG Evaluation Pipeline - Configuration${BLUE}      ┃${NC}"
echo -e "${BLUE}${BOLD}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
echo

dataset=$(get_user_choice "Select a dataset: " datasets)
chunking_strategy=$(get_user_choice "Select a chunking strategy: " chunking_strategies)
embedding_model=$(get_user_choice "Select an embedding model: " embedding_models)
top_k=$(get_user_choice "Select top_k value: " top_k_values)
enable_reranker=$(get_user_choice "Enable reranker? (yes/no): " enable_reranker_set)

echo ""
echo -e "${BLUE}${BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
echo -e "${BLUE}${BOLD}┃          ${MAGENTA}Selected Configuration Options${BLUE}          ┃${NC}"
echo -e "${BLUE}${BOLD}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
echo -e "${YELLOW}Dataset:${NC}           ${GREEN}$dataset${NC}"
echo -e "${YELLOW}Chunking strategy:${NC} ${GREEN}$chunking_strategy${NC}"
echo -e "${YELLOW}Embedding model:${NC}   ${GREEN}$embedding_model${NC}"
echo -e "${YELLOW}Top_k:${NC}             ${GREEN}$top_k${NC}"
echo -e "${YELLOW}Reranker:${NC}          ${GREEN}$enable_reranker${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e -n "${BOLD}${YELLOW}Are you sure? [Y/n] ${NC}"
read -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    cd "$SCRIPT_DIR"

    echo -e "\n${BLUE}${BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
    echo -e "${BLUE}${BOLD}┃              ${MAGENTA}Executing Evaluation${BLUE}                ┃${NC}"
    echo -e "${BLUE}${BOLD}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
    echo -e "${CYAN}▶ Working directory: ${GREEN}$(pwd)${NC}"
    echo -e "${CYAN}▶ Start time: ${GREEN}$(date '+%H:%M:%S')${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────${NC}"

    echo -e "${MAGENTA}${BOLD}Running evaluation with the following command:${NC}"
    echo -e "${CYAN}python -m src.main \\
    --corpus=\"${dataset}\" \\
    --chunker=\"${chunking_strategy}\" \\
    --embedding_model=\"${embedding_model}\" \\
    --top_k=\"${top_k}\" \\
    --reranker=\"${enable_reranker}\"${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────${NC}"
    echo ""

    python -m src.main \
        --corpus="${dataset}" \
        --chunker="${chunking_strategy}" \
        --embedding_model="${embedding_model}" \
        --top_k="${top_k}" \
        --reranker="${enable_reranker}"

    exit_status=$?

    echo -e "\n${BLUE}───────────────────────────────────────────────────${NC}"
    if [ $exit_status -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ Evaluation completed successfully!${NC}"
    else
        echo -e "${RED}${BOLD}✗ Evaluation failed with exit code ${exit_status}${NC}"
    fi
    echo -e "${CYAN}▶ End time: ${GREEN}$(date '+%H:%M:%S')${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
else
    echo -e "${YELLOW}Operation cancelled by user.${NC}"
fi