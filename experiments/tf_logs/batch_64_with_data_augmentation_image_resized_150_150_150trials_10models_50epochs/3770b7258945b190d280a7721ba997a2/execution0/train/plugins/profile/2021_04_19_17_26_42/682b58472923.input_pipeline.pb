	?d?VA'k@?d?VA'k@!?d?VA'k@	T;??$??T;??$??!T;??$??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?d?VA'k@????"`@1s?w??CU@A??%ǝұ?I?KK#@Yp?h?????*	6^?ILP?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??EB[n@!?	c5?X@)?ꐛ?&@1?o?I?V@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@yX?5?;??!?h!??^@)rM??΢??1?`(?z	@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@9
3???!s8%h?@)??bb?q??1p<s"?E@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?H??rڳ?!?h~O????)?H??rڳ?1?h~O????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?5?o????!?3b????)?5?o????1?3b????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????R???!W??ê??)?????Q??1c????:Preprocessing2F
Iterator::Modelx?ܙ	???!??}??r??)ʩ?ajK}?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9T;??$??Iږ??ON@Q?P?zؓC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????"`@????"`@!????"`@      ??!       "	s?w??CU@s?w??CU@!s?w??CU@*      ??!       2	??%ǝұ???%ǝұ?!??%ǝұ?:	?KK#@?KK#@!?KK#@B      ??!       J	p?h?????p?h?????!p?h?????R      ??!       Z	p?h?????p?h?????!p?h?????b      ??!       JGPUYT;??$??b qږ??ON@y?P?zؓC@