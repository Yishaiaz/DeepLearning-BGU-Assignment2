	Sz??=l@Sz??=l@!Sz??=l@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Sz??=l@߿yqka@1?;ۤU@A?d??]???Iq̲'? @*	?(\????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????k-6@!???"?X@)??߆?5@1^??T??X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@a???)??!`J؈?>??)?4?\????1
[?.3$??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchȱ?ᘱ?!(B?????)ȱ?ᘱ?1(B?????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??qѼ?!ܜ???,??)?0Bx?q??1G?_D?u??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??_?????!?#>????)??_?????1?#>????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??	Q??!4????)?3h?????1?/L??0??:Preprocessing2F
Iterator::Model?pz???!9kP????)˞6??y?1??]????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??<5OO@Q*(??ʰB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	߿yqka@߿yqka@!߿yqka@      ??!       "	?;ۤU@?;ۤU@!?;ۤU@*      ??!       2	?d??]????d??]???!?d??]???:	q̲'? @q̲'? @!q̲'? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??<5OO@y*(??ʰB@