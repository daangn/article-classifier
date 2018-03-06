RAILS_ROOT="$(pwd)/../hoian-webapp/tmp/cache"
docker run --rm -it --volumes-from gcloud-config \
         -v $(pwd):/app \
         -v "${RAILS_ROOT}/commercial_text/data/article.csv":/app/data/emb.csv \
         -v "${RAILS_ROOT}/commercial_text/data/text_normalized.txt.emb":/app/data/text_normalized.txt.emb \
         --entrypoint /bin/bash \
         daangn/cloud-ml:190
