	T???fDu@T???fDu@!T???fDu@	???}?????}??!???}??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-T???fDu@-y<-@1ۿ????t@I![??????Y?Ø?????*	??????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV29Dܜ?$@!?X?M?X@)???EC6$@1?????	X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@oe??2???!?????n@)t??%???1( ?????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@jhwH1??!??????)??uS?k??1?2?8?y??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@?cZ?????!???2ލ??)?cZ?????1???2ލ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?Cl?p???!ň}D.??)?Cl?p???1ň}D.??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???V???!?a??l??)??+?z???1D???????:Preprocessing2F
Iterator::Model?}:3P??!J???&Y??)?|????y?1I?^î?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???}??I??	.?O??Q?Զ??{X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-y<-@-y<-@!-y<-@      ??!       "	ۿ????t@ۿ????t@!ۿ????t@*      ??!       2      ??!       :	![??????![??????!![??????B      ??!       J	?Ø??????Ø?????!?Ø?????R      ??!       Z	?Ø??????Ø?????!?Ø?????b      ??!       JGPUY???}??b q??	.?O??y?Զ??{X@