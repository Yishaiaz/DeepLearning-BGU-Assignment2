	ŪA?[?e@ŪA?[?e@!ŪA?[?e@	Z??T????Z??T????!Z??T????"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ŪA?[?e@A?} R?
@1܀???d@Io?$????Y$)?ahu??*	??Q????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??????(@!?k?T?X@)Zf?? (@1q?HvX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@@i?QH2??!?8?@)?pvk???1???(????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismX??????!??>????)<Mf?????1͸]z???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?a?7?W??!F!?H???)???,z??1?!7?^??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?????5??!veZJ??)?????5??1veZJ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchs??c?Ȱ?!??tj,???)s??c?Ȱ?1??tj,???:Preprocessing2F
Iterator::ModeloF?W????!?e|Ϊ??)????x!}?1I??g???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Z??T????I ?B??@Q?	q(2*X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	A?} R?
@A?} R?
@!A?} R?
@      ??!       "	܀???d@܀???d@!܀???d@*      ??!       2      ??!       :	o?$????o?$????!o?$????B      ??!       J	$)?ahu??$)?ahu??!$)?ahu??R      ??!       Z	$)?ahu??$)?ahu??!$)?ahu??b      ??!       JGPUYZ??T????b q ?B??@y?	q(2*X@