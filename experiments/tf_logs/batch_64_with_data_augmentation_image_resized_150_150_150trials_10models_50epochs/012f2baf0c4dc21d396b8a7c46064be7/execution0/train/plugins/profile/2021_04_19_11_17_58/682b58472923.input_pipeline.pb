	G6?:h@G6?:h@!G6?:h@	5?k?]???5?k?]???!5?k?]???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6G6?:h@??]/MRZ@1???t?bU@A"??ƽ???IIط??? @Y?I??	???*	????&??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??6ʚ@!?????X@)??7?@1H?s?z"W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@V??#)??!????@)???????1?Hi7?@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@yY|E??!?.ľ@)??vۅ???1B??RmY??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@m?_u?H??!Wv?*????)m?_u?H??1Wv?*????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???덪?!??????)???덪?1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??J?????! db?u??)#??)?ϖ?1???}J???:Preprocessing2F
Iterator::Model???);???!??R?B??)?I?????1f8ք7h??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no95?k?]???Ip??ƷK@Q@Z??F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??]/MRZ@??]/MRZ@!??]/MRZ@      ??!       "	???t?bU@???t?bU@!???t?bU@*      ??!       2	"??ƽ???"??ƽ???!"??ƽ???:	Iط??? @Iط??? @!Iط??? @B      ??!       J	?I??	????I??	???!?I??	???R      ??!       Z	?I??	????I??	???!?I??	???b      ??!       JGPUY5?k?]???b qp??ƷK@y@Z??F@