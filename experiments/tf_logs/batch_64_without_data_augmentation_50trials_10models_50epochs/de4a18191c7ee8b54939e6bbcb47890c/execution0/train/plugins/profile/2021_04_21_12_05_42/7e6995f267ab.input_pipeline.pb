	8?0C?"u@8?0C?"u@!8?0C?"u@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-8?0C?"u@?9?}??	@1q????t@Al@??r??I'i???F@*	?? ???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2p???$D&@!?Ґwn?X@)?????%@1pWu??2X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??q?????!\fo?r@)*U??-???1?6A?W??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?̒ 5???!ĕ??????)?%s,荒?1?^91D???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@\t??z???!??lo???)\t??z???1??lo???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchYM?]??!{٩?????)YM?]??1{٩?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?W:?%??!4?6(???)|b?*?3??1s%?.?@??:Preprocessing2F
Iterator::Model62;?޹?!Ů?7????)??z?ю{?1)?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@xAs?U??Q?2~??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9?}??	@?9?}??	@!?9?}??	@      ??!       "	q????t@q????t@!q????t@*      ??!       2	l@??r??l@??r??!l@??r??:	'i???F@'i???F@!'i???F@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@xAs?U??y?2~??X@