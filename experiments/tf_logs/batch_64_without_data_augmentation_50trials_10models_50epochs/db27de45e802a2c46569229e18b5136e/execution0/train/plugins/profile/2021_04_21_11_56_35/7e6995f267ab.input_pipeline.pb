	?UJ?tu@?UJ?tu@!?UJ?tu@      ??!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?UJ?tu@cD?в.	@14?Y?w?t@I?h?'?O@*	/?$?U?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?>V?۰%@!???:?X@)3m??J#%@1gy?;0X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?3ڪ$???!;??Q??@)W|C??u??1?Z+~?H??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@P?eo)??!?A?%C7??)i7????1n?.???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@7+1?J??!?n????)7+1?J??1?n????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????X???!l> ????)????X???1l> ????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??n?;2??!??$?~???)????=z??1??Q??I??:Preprocessing2F
Iterator::Modelr?)????!???????)????|?1E??WB???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?Y?lA/??Q?
L?B?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	cD?в.	@cD?в.	@!cD?в.	@      ??!       "	4?Y?w?t@4?Y?w?t@!4?Y?w?t@*      ??!       2      ??!       :	?h?'?O@?h?'?O@!?h?'?O@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Y?lA/??y?
L?B?X@