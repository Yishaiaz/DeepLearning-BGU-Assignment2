	?|??e@?|??e@!?|??e@	$4h???$4h???!$4h???"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?|??e@8???C@1Z????e@IR&5???Y???????*	????B?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2Ov3??$@!x??y?X@)??U?P	$@1ޤ???%X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??
DO???!3?:<@)?!?{???1W??????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?R%??R??!?:?=~??)?L!u??1n1?E??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@!???0??!D33f???)!???0??1D33f???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????g???!??a????)????g???1??a????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismط???/??!y??.???)!?Ky ??1.@ x???:Preprocessing2F
Iterator::Model[?}s??!C??>C??)$H???8t?1?k??^??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9$4h???I?9?91?@Q??cɚ!X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8???C@8???C@!8???C@      ??!       "	Z????e@Z????e@!Z????e@*      ??!       2      ??!       :	R&5???R&5???!R&5???B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JGPUY$4h???b q?9?91?@y??cɚ!X@