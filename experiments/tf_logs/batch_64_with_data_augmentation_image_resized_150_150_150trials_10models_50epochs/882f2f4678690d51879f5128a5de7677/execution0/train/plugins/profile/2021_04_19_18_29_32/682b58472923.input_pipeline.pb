	??zO??h@??zO??h@!??zO??h@	?h?+A???h?+A??!?h?+A??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??zO??h@?????[@1??g?"U@A????y??I/??)@Y?_cD???*	??ʁm?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??I~?@!e+n???X@)̘?5>@1qʼM?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@h^????!9?e'@)????K???1y???Z?	@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@O?}????!?<|}pT@)S?Q?G??1?'pD?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@K?????!?*r¥??)K?????1?*r¥??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA?)V¬?!ӱh?$??)A?)V¬?1ӱh?$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismv?d??!m6????)WAt???1?ƦF????:Preprocessing2F
Iterator::Model
?s34??!?&u????)9??U}?1W??.R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?h?+A??I?n??_L@QjG??i?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????[@?????[@!?????[@      ??!       "	??g?"U@??g?"U@!??g?"U@*      ??!       2	????y??????y??!????y??:	/??)@/??)@!/??)@B      ??!       J	?_cD????_cD???!?_cD???R      ??!       Z	?_cD????_cD???!?_cD???b      ??!       JGPUY?h?+A??b q?n??_L@yjG??i?E@