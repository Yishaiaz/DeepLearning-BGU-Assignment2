	?{?5Zif@?{?5Zif@!?{?5Zif@	u߮????u߮????!u߮????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?{?5Zif@?)???@1/???u?e@I?Cl?p???Y5E?ӻx??*	|?5^zj?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV28???'@!O?Xz?X@)?ǚ?A&'@18???"X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?Χ?U??!?<?Us3@)?,??2??1??????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@???????!?y0?cX??)?30??&??1?;Sfk??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@kdWZF???!soܷ0??)kdWZF???1soܷ0??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?7? ?ث?!?2????)?7? ?ث?1?2????:Preprocessing2F
Iterator::Model?̒ 5???!eqX?????)?.?.??1???&=??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?GW????!-??]????)O?\?	??1??Zf????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9u߮????I ????@Q??$-,X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)???@?)???@!?)???@      ??!       "	/???u?e@/???u?e@!/???u?e@*      ??!       2      ??!       :	?Cl?p????Cl?p???!?Cl?p???B      ??!       J	5E?ӻx??5E?ӻx??!5E?ӻx??R      ??!       Z	5E?ӻx??5E?ӻx??!5E?ӻx??b      ??!       JGPUYu߮????b q ????@y??$-,X@