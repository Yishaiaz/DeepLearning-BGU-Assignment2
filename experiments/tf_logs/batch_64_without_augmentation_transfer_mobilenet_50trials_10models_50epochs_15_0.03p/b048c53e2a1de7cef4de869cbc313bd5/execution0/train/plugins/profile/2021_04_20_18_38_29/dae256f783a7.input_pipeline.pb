	???(k@???(k@!???(k@	ë`? ??ë`? ??!ë`? ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???(k@?qnnL`@1?BX?%'U@A?(^emS??I.??'Hl??YW|C??u??*	?Ve?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?%?<?6@!֗?R?X@)?	?8?6@1?l9???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?(_?B??!!O??[??)???w?̸?1㰗?	??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@/3l?????!????v2??)??+H3??1?B?????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@D?X?oC??!?G????)D?X?oC??1?G????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchx?????!0f`v???)x?????10f`v???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?YJ??P??!????)"??<???1KυF?ϫ?:Preprocessing2F
Iterator::Model4??7?¬?!
?S??Z??)?CR%?s?16x?BW??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ë`? ??I???EN@Q&-v>?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?qnnL`@?qnnL`@!?qnnL`@      ??!       "	?BX?%'U@?BX?%'U@!?BX?%'U@*      ??!       2	?(^emS???(^emS??!?(^emS??:	.??'Hl??.??'Hl??!.??'Hl??B      ??!       J	W|C??u??W|C??u??!W|C??u??R      ??!       Z	W|C??u??W|C??u??!W|C??u??b      ??!       JGPUYë`? ??b q???EN@y&-v>?C@