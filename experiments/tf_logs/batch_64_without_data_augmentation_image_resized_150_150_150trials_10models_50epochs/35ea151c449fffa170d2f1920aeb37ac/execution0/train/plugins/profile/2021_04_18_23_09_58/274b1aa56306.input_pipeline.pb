	?U?@X?f@?U?@X?f@!?U?@X?f@	/??<e???/??<e???!/??<e???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?U?@X?f@???u??V@1???V@A???W???I??ԱJI@YyW=`2??*	?????Ȯ@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????k?@!>?Mp?wX@)
j??m@1,/p?ӋV@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?+???d??!)A???@)|?/????1S??0-@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??Ϸ??!?z?@)?Rz????1???(Q@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@V?P?????!?|????)V?P?????1?|????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch2??z?p??!??͆M??)2??z?p??1??͆M??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??4?(??!^?E?????)W?9?m?1ĕ??????:Preprocessing2F
Iterator::Model4w??o??!+?M?1 @)h?o}Xot?1????4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no90??<e???In??x??I@Q??'"?;H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???u??V@???u??V@!???u??V@      ??!       "	???V@???V@!???V@*      ??!       2	???W??????W???!???W???:	??ԱJI@??ԱJI@!??ԱJI@B      ??!       J	yW=`2??yW=`2??!yW=`2??R      ??!       Z	yW=`2??yW=`2??!yW=`2??b      ??!       JGPUY0??<e???b qn??x??I@y??'"?;H@