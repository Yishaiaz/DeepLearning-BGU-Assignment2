	`vOKk@`vOKk@!`vOKk@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-`vOKk@wj.7??`@1?????$U@A??9?ؗ??I??U???*	NbX??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?pt??&6@!??	?x?X@)??L??5@1py???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@5?;???!???c???){?\?&???1???o?:??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??!p??!?"?????)???_?5??1???@~??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??RB????!'?	?????)??RB????1'?	?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch׈`\:??!? wgU??)׈`\:??1? wgU??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism؀q????!䉖]b???)??R%ʎ?1??k??R??:Preprocessing2F
Iterator::Model?K??$w??!?o ?G???)иp $x?1?]??Y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI? ?_ݡN@Q?4?"^C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	wj.7??`@wj.7??`@!wj.7??`@      ??!       "	?????$U@?????$U@!?????$U@*      ??!       2	??9?ؗ????9?ؗ??!??9?ؗ??:	??U?????U???!??U???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? ?_ݡN@y?4?"^C@