	9{g?պk@9{g?պk@!9{g?պk@	W??????W??????!W??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69{g?պk@?
?rG`@1????%V@A)???^??Iv?TQ@Y???2??*	?O??n??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????@!??;??X@)=?බ @1Ta?@yW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@ ??????!?Td?]C@)@k~??E??1?????m@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@???A???!??[?	@)?|?b?:??1?{?n?p??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@àL??Ű?!D?I????)àL??Ű?1D?I????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???m??!?́V???)???m??1?́V???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismd??u??!O??N???)>???6??1?!?4?k??:Preprocessing2F
Iterator::ModelܷZ'.ǳ?!?Wq9???)X<?H??z?1?	??T7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W??????IF?,??M@Q???Nl?C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?
?rG`@?
?rG`@!?
?rG`@      ??!       "	????%V@????%V@!????%V@*      ??!       2	)???^??)???^??!)???^??:	v?TQ@v?TQ@!v?TQ@B      ??!       J	???2?????2??!???2??R      ??!       Z	???2?????2??!???2??b      ??!       JGPUYW??????b qF?,??M@y???Nl?C@