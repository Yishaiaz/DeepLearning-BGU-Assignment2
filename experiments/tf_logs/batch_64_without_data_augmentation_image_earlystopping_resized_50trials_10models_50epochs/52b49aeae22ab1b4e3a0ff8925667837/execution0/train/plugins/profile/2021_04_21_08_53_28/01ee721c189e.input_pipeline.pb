	{3j?J_@{3j?J_@!{3j?J_@	
?x?z??
?x?z??!
?x?z??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{3j?J_@?g???-A@1?+?V]?U@A2?CP5??I?ҿ$?)@Yd?mlv$??*	L7?A@˱@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?j,a?@![&mӘX@)??4??@1???+W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??w????!?????@)??!????1?&آ?C@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@;ŪA??!?E?:????);ŪA??1?E?:????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?????!??V??s@)h%???³?1?%@???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch닄??K??!&G??hZ??)닄??K??1&G??hZ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism>?N????!;?M?v??)!?1??Б?1??7h:q??:Preprocessing2F
Iterator::Model?a̲ۢ?!ki??$???)??0a4+{?1?r	rk???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9	?x?z??I??t???=@Q???(1_Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?g???-A@?g???-A@!?g???-A@      ??!       "	?+?V]?U@?+?V]?U@!?+?V]?U@*      ??!       2	2?CP5??2?CP5??!2?CP5??:	?ҿ$?)@?ҿ$?)@!?ҿ$?)@B      ??!       J	d?mlv$??d?mlv$??!d?mlv$??R      ??!       Z	d?mlv$??d?mlv$??!d?mlv$??b      ??!       JGPUY	?x?z??b q??t???=@y???(1_Q@