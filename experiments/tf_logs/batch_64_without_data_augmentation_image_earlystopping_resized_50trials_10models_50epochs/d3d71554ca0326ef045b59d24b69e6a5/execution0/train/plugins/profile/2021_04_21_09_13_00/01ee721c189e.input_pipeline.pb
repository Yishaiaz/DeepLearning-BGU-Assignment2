	whX??oZ@whX??oZ@!whX??oZ@	?OP??????OP?????!?OP?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6whX??oZ@e73???0@1]p??U@A4?l\??IAaP??$??Y???j?=??*	?I?Ư@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2>?4a??@!?=??!?X@)Y??w @1,S=WHV@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@(??h????!?L???@)(??h????1?L???@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@m???L??!=QU?!@)Q?+?Ͼ?1??j«@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@2Xq??0??!?jR???@)??N??1??X?( @:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchND??~???!|j?Nl??)ND??~???1|j?Nl??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??J̳???!?y+?;???)?????U??1?Ƃ??;??:Preprocessing2F
Iterator::Model??N?`???!.???????)S?A?Ѫv?1Β(?fj??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?OP?????I?????1@Q?q3(ImT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e73???0@e73???0@!e73???0@      ??!       "	]p??U@]p??U@!]p??U@*      ??!       2	4?l\??4?l\??!4?l\??:	AaP??$??AaP??$??!AaP??$??B      ??!       J	???j?=?????j?=??!???j?=??R      ??!       Z	???j?=?????j?=??!???j?=??b      ??!       JGPUY?OP?????b q?????1@y?q3(ImT@