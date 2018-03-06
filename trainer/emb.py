LABEL_COL = 0
ID_COL = 1
IMAGES_COUNT_COL = 4

def get_shard(id):
    return id / 10000

def id_to_path(id):
    shard = get_shard(id)
    return "%d/%d.emb" % (shard, id)

