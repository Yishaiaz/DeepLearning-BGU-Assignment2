	??6S?i@??6S?i@!??6S?i@	?(+j????(+j???!?(+j???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??6S?i@Z????Z@1?q?
aV@A,*?t????I4????? @Y?c?????*	?O??n?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2"7???@!?|@6?X@)????@?@16?bk?W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?X5s???!~n?Q?X@)???Im??1?qT??@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@b?k_@??!?-?Nd?
@)?|??z???1mF??<???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@???&????!e??"??)???&????1e??"??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ګ????!?{Ho??)?ګ????1?{Ho??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismV??y???!????3??)ZՒ?r0??1*tw?????:Preprocessing2F
Iterator::Model???
~??!????o2??)?&??d?v?1?Q?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?(+j???IL{?i?oK@Q?2???TF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z????Z@Z????Z@!Z????Z@      ??!       "	?q?
aV@?q?
aV@!?q?
aV@*      ??!       2	,*?t????,*?t????!,*?t????:	4????? @4????? @!4????? @B      ??!       J	?c??????c?????!?c?????R      ??!       Z	?c??????c?????!?c?????b      ??!       JGPUY?(+j???b qL{?i?oK@y?2???TF@