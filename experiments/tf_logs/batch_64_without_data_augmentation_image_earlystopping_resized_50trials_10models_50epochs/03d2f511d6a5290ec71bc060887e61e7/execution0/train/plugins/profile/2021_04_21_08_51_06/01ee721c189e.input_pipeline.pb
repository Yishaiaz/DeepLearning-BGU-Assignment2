	[??Y~^@[??Y~^@![??Y~^@	ڍt?????ڍt?????!ڍt?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6[??Y~^@?#??tA@1C=}?9U@Ao??}U.??I?uii??Y?$>w?=??*	?z?gа@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?U???@!??lv??X@)??=x?@1m?iW?-W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@UQ??ڦ??!?75??@@)????%ƺ?1u?Zp@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@????C??!??h??	@)y??[Y???1XP(pC???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@
K<?lʭ?!BG??ɠ??)
K<?lʭ?1BG??ɠ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchOʤ?6 ??!??TE???)Oʤ?6 ??1??TE???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?I?p??!{???xR??)??D.8???16?????:Preprocessing2F
Iterator::Model??\????!8??db???)8i?x?1??vpNo??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ڍt?????IX??*?t=@QNhy[gQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#??tA@?#??tA@!?#??tA@      ??!       "	C=}?9U@C=}?9U@!C=}?9U@*      ??!       2	o??}U.??o??}U.??!o??}U.??:	?uii???uii??!?uii??B      ??!       J	?$>w?=???$>w?=??!?$>w?=??R      ??!       Z	?$>w?=???$>w?=??!?$>w?=??b      ??!       JGPUYڍt?????b qX??*?t=@yNhy[gQ@