	?A'??gl@?A'??gl@!?A'??gl@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?A'??gl@w?k?F?a@1h@?5.U@A<??kз?I??x?????*	?????|?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2\u?)?6@!??lu?X@)B???6@1OP??_?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@ (??{???!d????
??)??QF\ ??1o??I????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?d?<??!??k<?x??)?d?<??1??k<?x??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@W\?????!.?Ug??)?9?!??1RS???s??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchi?"?~??!v?Y????)i?"?~??1v?Y????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismN?#Ed??!@?????)f?YJ????1??????:Preprocessing2F
Iterator::Model?/??????!{?1????)uۈ'?y?1?3?I@???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?y?[O@Q?1??O?B@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?k?F?a@w?k?F?a@!w?k?F?a@      ??!       "	h@?5.U@h@?5.U@!h@?5.U@*      ??!       2	<??kз?<??kз?!<??kз?:	??x???????x?????!??x?????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?y?[O@y?1??O?B@