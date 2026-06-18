products_filepath="data/products-large.jsonl"
documents_filepath="resources/documents.jsonl"
index_input_dir="${INDEX_INPUT_DIR:-resources/index_input}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-8}"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xmx8g -XX:MaxRAMPercentage=25}"

## convert products to documents
#rm -rf resources
#mkdir -p resources
#python src/search_engine/convert_products_to_documents.py $products_filepath $documents_filepath
#if [ $? -eq 0 ]; then
#    echo "convert products to documents success"
#else
#    exit 1
#fi

# build indexes
rm -rf indexes
mkdir -p indexes
rm -rf "$index_input_dir"
mkdir -p "$index_input_dir"
ln -s "$(realpath "$documents_filepath")" "$index_input_dir/documents.jsonl"
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$index_input_dir" \
  --index indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
if [ $? -eq 0 ]; then
    echo "build indexes success"
else
    exit 1
fi
