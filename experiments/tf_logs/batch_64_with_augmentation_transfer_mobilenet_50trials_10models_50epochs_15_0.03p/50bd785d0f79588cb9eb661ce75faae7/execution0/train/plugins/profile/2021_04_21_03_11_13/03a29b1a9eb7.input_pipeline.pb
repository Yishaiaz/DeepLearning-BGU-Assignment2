	????0m@????0m@!????0m@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????0m@???^b@1!yv?U@A??E	???I?????? @*	??? `??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2H?3?9Y5@!e?d?X@)?k???(5@1b???˲X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@H???=??!2?#nL??)0??mP???1???З???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@`#I????!&U????)n?8)?{??1?68????:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@QhY?????!h???????)QhY?????1h???????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchy???h??!~?:A~??)y???h??1~?:A~??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?uS?k%??!?4?"????)e?,?i???1???f??:Preprocessing2F
Iterator::Model?tx㧱?!y??=????)??O?s'x?1k???1??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 62.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI"MЯ??O@Q޲/PmB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???^b@???^b@!???^b@      ??!       "	!yv?U@!yv?U@!!yv?U@*      ??!       2	??E	?????E	???!??E	???:	?????? @?????? @!?????? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q"MЯ??O@y޲/PmB@