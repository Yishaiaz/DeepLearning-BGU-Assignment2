	?B?n6h@?B?n6h@!?B?n6h@	f;????f;????!f;????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?B?n6h@?p!??Y@1[???X#V@A7o??=??I??'?.???Y	???W??*	?????g?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2K?ó?@!Pޮm?X@)jm?k?@1.?.?H?V@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?R???!?????@)1
?Ƿw??1??'?%?@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@c	kc????!3???M@)?qp?????1H?????@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?Ss??P??!y
?? M??)?Ss??P??1y
?? M??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??5?K??!?*}P?U??)??5?K??1?*}P?U??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismmscz???!?)C???);Qi??1???n%??:Preprocessing2F
Iterator::Model*U??-???!??xH?$??)9?	?ʼu?1???,??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9f;????I?fk?J@Qmy???F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?p!??Y@?p!??Y@!?p!??Y@      ??!       "	[???X#V@[???X#V@![???X#V@*      ??!       2	7o??=??7o??=??!7o??=??:	??'?.?????'?.???!??'?.???B      ??!       J		???W??	???W??!	???W??R      ??!       Z		???W??	???W??!	???W??b      ??!       JGPUYf;????b q?fk?J@ymy???F@