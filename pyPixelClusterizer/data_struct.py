# Data structure definitions to be used by numpy within python

import numpy as np

HitInfo = np.dtype([('event_number', '<i8'),
                    ('frame', '<u1'),
                    ('column', '<u2'),
                    ('row', '<u2'),
                    ('charge', '<u2')])

ClusterHitInfo = np.dtype([('event_number', '<i8'),
                           ('frame', '<u1'),
                           ('column', '<u2'),
                           ('row', '<u2'),
                           ('charge', '<u2'),
                           ('cluster_ID', '<u2'),
                           ('is_seed', '<u1'),
                           ('cluster_size', '<u2'),
                           ('n_cluster', '<u2')])


ClusterInfo = np.dtype([('event_number', '<i8'),
                        ('ID', '<u2'),
                        ('size', '<u2'),
                        ('charge', '<u2'),
                        ('seed_column', '<u2'),
                        ('seed_row', '<u2'),
                        ('mean_column', 'f4'),
                        ('mean_row', 'f4')])
