	?j?3	l@?j?3	l@!?j?3	l@	9­*N??9­*N??!9­*N??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?j?3	l@?}???`@1??G7?U@A?~?:pΨ?I@x?=? @Y?ڋh;???*	?????"?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2>[?6@!9??X@)???-?g6@1[t(?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?-?l???!?I:??1??)??K⬸?1[?S?6??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?Y?H?s??!?? ??,??)?E?~???1???-n???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@mU???!J	?"A???)mU???1J	?"A???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?'c|????!?p=????)?'c|????1?p=????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?f?ba???!?ة?x??)??8?#+??1??s?0??:Preprocessing2F
Iterator::Model#. ?ҥ??!wyM?s??):vP??x?1?PB?|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9:­*N??I???(dnN@Q??:?M~C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?}???`@?}???`@!?}???`@      ??!       "	??G7?U@??G7?U@!??G7?U@*      ??!       2	?~?:pΨ??~?:pΨ?!?~?:pΨ?:	@x?=? @@x?=? @!@x?=? @B      ??!       J	?ڋh;????ڋh;???!?ڋh;???R      ??!       Z	?ڋh;????ڋh;???!?ڋh;???b      ??!       JGPUY:­*N??b q???(dnN@y??:?M~C@