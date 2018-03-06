import os
import sys
import glob
import logging
from subprocess import call
from multiprocessing import Pool

import click
from tensorflow.python.lib.io import file_io

from trainer.emb import get_shard, id_to_path, ID_COL, IMAGES_COUNT_COL

emb_path = 'data/image_embeddings'
gs_emb_path = 'gs://towneers-ml/article_classifier'
sync_cloud = False

def remove(id, cloud_path=None):
    filepath = "%s/%s" % (emb_path, id_to_path(id))
    if cloud_path:
        cloud_filepath = '%s/%s' % (cloud_path, filepath)
        if file_io.file_exists(cloud_filepath):
            file_io.delete_file(cloud_filepath)
    if file_io.file_exists(filepath):
        return file_io.delete_file(filepath)

def down(id, cloud_path=None):
    shard = get_shard(id)
    to_path = "%s/%d" % (emb_path, shard)
    if not os.path.exists(to_path):
        try:
            os.mkdir(to_path)
        except OSError as e:
            if e.errno != 17: # File exists
                raise e
    to_filepath = "%s/%d.emb" % (to_path, id)
    url = 'http://ml.daangn.com/articles/image_embeddings/%s' % id_to_path(id)
    logging.info('down: %s', url)
    result = call(['curl', '-f', '--connect-timeout', '2', '-o', to_filepath, url])
    if not os.path.exists(to_filepath):
        return 0
    if os.stat(to_filepath).st_size < 1:
        os.remove(filepath)
        return 0

    if cloud_path:
        to_gs_filepath = '%s/%s' % (cloud_path, to_filepath)
        if file_io.file_exists(to_gs_filepath):
            return 0

        to_gs_path = '%s/%s' % (cloud_path, to_path)
        if not file_io.is_directory(to_gs_path):
            file_io.create_dir(to_gs_path)
        file_io.copy(to_filepath, to_gs_filepath)
    return 1

def remove_cloud(id):
    remove(id, gs_emb_path)

def down_cloud(id):
    down(id, gs_emb_path)

@click.command()
@click.option('--cloud', is_flag=True, help='sync cloud')
def main(cloud):
    with open('data/emb.csv') as f:
        rows = [line.split(',') for line in f.readlines()[1:]]
        new_ids = set([int(row[ID_COL]) for row in rows if row[IMAGES_COUNT_COL] != '0'])
    print 'new ids count:', len(new_ids)
    if cloud:
        print 'sync cloud'

    if cloud:
        pattern = '%s/%s/*/*.emb' % (gs_emb_path, emb_path)
        pathes = file_io.get_matching_files(pattern)
    else:
        pattern = '%s/*/*.emb' % emb_path
        pathes = glob.glob(pattern)

    local_ids = set([int(x.split('/')[-1][:-4]) for x in pathes])
    print 'local ids count:', len(local_ids)

    if len(new_ids) < 1:
        raise Exception('new_ids is zero')

    add_ids = new_ids - local_ids
    remove_ids = local_ids - new_ids

    print "add ids count: %d" % len(add_ids)
    print "remove ids count: %d" % len(remove_ids)

    p = Pool()

    print 'removing count: %d' % len(remove_ids)
    if cloud:
        results = p.map(remove_cloud, remove_ids)
    else:
        results = p.map(remove, remove_ids)
    print 'local files removed'

    print 'downloading %d files' % len(add_ids)
    if len(add_ids) < 1000:
        if cloud:
            results = p.map(down_cloud, add_ids)
        else:
            results = p.map(down, add_ids)
        print 'downloaded count: %d / %d' % (sum(results), len(results))
    else:
        with open('data/down_emb_files.txt', 'w') as f:
            for id in add_ids:
                filepath = id_to_path(id)
                f.write("%s\n" % filepath)
            print '%d files list saved to %s' % (len(add_ids), f.name)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
