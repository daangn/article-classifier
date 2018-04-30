RAILS_ROOT="$(pwd)/../hoian-webapp/tmp/cache/commercial_text/data"
docker run --rm -it --volumes-from gcloud-config \
         -v $(pwd):/app \
         -v "${RAILS_ROOT}/article.csv":/app/data/emb.csv \
         -v "${RAILS_ROOT}/title_normalized.txt.emb":/app/data/title_normalized.txt.emb \
         -v "${RAILS_ROOT}/content_normalized.txt.emb":/app/data/content_normalized.txt.emb \
         -v "${RAILS_ROOT}/title_normalized.txt.emb.words":/app/data/title_normalized.txt.emb.words \
         -v "${RAILS_ROOT}/content_normalized.txt.emb.words":/app/data/content_normalized.txt.emb.words \
         --entrypoint /bin/bash \
         daangn/cloud-ml:199
