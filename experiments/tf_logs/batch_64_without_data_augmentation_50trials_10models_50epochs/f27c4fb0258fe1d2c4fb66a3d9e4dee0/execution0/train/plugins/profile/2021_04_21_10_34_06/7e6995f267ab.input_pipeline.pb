	??S??t@??S??t@!??S??t@	??Q??????Q????!??Q????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??S??t@?;ۤ"@1??)X#~t@I??}??@Y? ??%s??*	`??"???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??^D?!$@!?K.?9?X@)??l?#@1??,@X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@????????!s??5@)[@h=|???1?=4i??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??eO??!?Q7???)=?????1??	4???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?)??s??!?$?:&??)?)??s??1?$?:&??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch^K?=???!xą?3??)^K?=???1xą?3??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismJ?>?ɷ?!Y~?|@??)5]Ot]???1??????:Preprocessing2F
Iterator::Model8,?????!???c??)?~???{?1X|l`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Q????I ŨE??Q????|X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?;ۤ"@?;ۤ"@!?;ۤ"@      ??!       "	??)X#~t@??)X#~t@!??)X#~t@*      ??!       2      ??!       :	??}??@??}??@!??}??@B      ??!       J	? ??%s??? ??%s??!? ??%s??R      ??!       Z	? ??%s??? ??%s??!? ??%s??b      ??!       JGPUY??Q????b q ŨE??y????|X@