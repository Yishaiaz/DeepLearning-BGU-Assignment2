	??B=?e@??B=?e@!??B=?e@	???ǚ??????ǚ???!???ǚ???"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??B=?e@??2P@1?_?5?/e@IqX?Q??Y7???????*	?v??
?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?X??i'@!?p???X@)??Û5?&@1NT?JFX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@ C?*??!??#?? @)3?뤾,??1
? bY???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch ??q????!+{ܫ??) ??q????1+{ܫ??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??0???!?{?/Y??)??0???1?{?/Y??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?Q<????!??6????)??;jL???1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???,յ?!????"??)??4)ݎ?19G?'5Z??:Preprocessing2F
Iterator::Model$?@??!U??(?q??)?7L4H?s?1?I0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???ǚ???I ??3~@Q???s?/X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??2P@??2P@!??2P@      ??!       "	?_?5?/e@?_?5?/e@!?_?5?/e@*      ??!       2      ??!       :	qX?Q??qX?Q??!qX?Q??B      ??!       J	7???????7???????!7???????R      ??!       Z	7???????7???????!7???????b      ??!       JGPUY???ǚ???b q ??3~@y???s?/X@