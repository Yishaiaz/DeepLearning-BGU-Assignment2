	3o?u??t@3o?u??t@!3o?u??t@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-3o?u??t@Iط?	@1?Xİ?st@A??Z?a/d?I??7??d@*	??~jLa?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?QF\ ?$@!??}??X@)Y??;!$@1?u?<@X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@b?qm???!??F??z@)????2??1?45???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??? ³?!??n~,???)??? ³?1??n~,???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?j????!?vX?]???)4?IbI???1?{?s????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?4c?tv??!zq%c ??)?4c?tv??1zq%c ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism]N	?I???!u݁E????)??d#ٓ?1l?K????:Preprocessing2F
Iterator::Model?ǚ?A??!?)?t??)Y??+??x?1	?}?r??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI??:?}???Q?	n?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Iط?	@Iط?	@!Iط?	@      ??!       "	?Xİ?st@?Xİ?st@!?Xİ?st@*      ??!       2	??Z?a/d???Z?a/d?!??Z?a/d?:	??7??d@??7??d@!??7??d@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??:?}???y?	n?X@