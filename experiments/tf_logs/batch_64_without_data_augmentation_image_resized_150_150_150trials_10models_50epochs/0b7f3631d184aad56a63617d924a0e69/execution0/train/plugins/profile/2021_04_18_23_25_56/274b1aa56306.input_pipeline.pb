	`?;?	i@`?;?	i@!`?;?	i@	???M<5?????M<5??!???M<5??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6`?;?	i@x?q?Z?Z@1???Dh{V@A?/EHݲ?I?? @Y?l??3H??*	_?I??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2???Bt8@!tP?Qr?X@)g??6@1y???SW@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@\??&??!??q#?@)j?????1??o?%? @:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?Ӂ??V??!?k?:!?@)??׻??1a??ԼF??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@i? ?w???!?B???C??)i? ?w???1?B???C??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch_?L?J??!p?Y?w???)_?L?J??1p?Y?w???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!Vo???m??)?R?!?u??1?C{??\??:Preprocessing2F
Iterator::Modelv??^
??!??k?kc??)tys?V{x?1287[?[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???M<5??IդSK@Q;?$?dF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	x?q?Z?Z@x?q?Z?Z@!x?q?Z?Z@      ??!       "	???Dh{V@???Dh{V@!???Dh{V@*      ??!       2	?/EHݲ??/EHݲ?!?/EHݲ?:	?? @?? @!?? @B      ??!       J	?l??3H???l??3H??!?l??3H??R      ??!       Z	?l??3H???l??3H??!?l??3H??b      ??!       JGPUY???M<5??b qդSK@y;?$?dF@