	?ŦUfu@?ŦUfu@!?ŦUfu@	?lZJ?C???lZJ?C??!?lZJ?C??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ŦUfu@?9@0G?@1F?W???t@AuV?1???I?W?????Y???{???*	a??"?_?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV25&?\R?)@!??ߒ??X@)a??L&)@1?X?h2X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@{??????!i}(?	@)???4????1?k:QT???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@am?????!?f??d???)am?????1?f??d???:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@@a??+??!4???$??)?K?K?1??1?????Z??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?? ?X4??!????f??)?? ?X4??1????f??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?)??ѫ?!T?????)????Fu??1	????t??:Preprocessing2F
Iterator::Model??Po??!?tc mH??)?뤾,?t?1???mj"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?lZJ?C??I??¢??Qe];1?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?9@0G?@?9@0G?@!?9@0G?@      ??!       "	F?W???t@F?W???t@!F?W???t@*      ??!       2	uV?1???uV?1???!uV?1???:	?W??????W?????!?W?????B      ??!       J	???{??????{???!???{???R      ??!       Z	???{??????{???!???{???b      ??!       JGPUY?lZJ?C??b q??¢??ye];1?X@