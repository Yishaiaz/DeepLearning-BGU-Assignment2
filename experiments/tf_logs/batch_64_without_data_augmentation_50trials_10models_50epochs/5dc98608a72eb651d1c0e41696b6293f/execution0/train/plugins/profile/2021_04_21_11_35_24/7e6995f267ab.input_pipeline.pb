	????v?t@????v?t@!????v?t@	?:'??M???:'??M??!?:'??M??"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????v?t@d?M*K@1??t??t@I)?k{?? @Y?D?+??*	y?&1?W?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?V????(@!?7??G?X@)?"1A(@1?Z.!X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?z?p̲??!??5&?@)???`U??1R?NCpb??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??g\8??!???&?#??)??)1	??1???g???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??????!W0??'*??)??????1W0??'*??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Ց#??!"> |0??)??Ց#??1"> |0??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????1v??! ?????)???J??1??Q??Z??:Preprocessing2F
Iterator::Model?!??I??!+?(:\??);?vٯ;}?1?P??Q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?:'??M??I???????Q8o?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d?M*K@d?M*K@!d?M*K@      ??!       "	??t??t@??t??t@!??t??t@*      ??!       2      ??!       :	)?k{?? @)?k{?? @!)?k{?? @B      ??!       J	?D?+???D?+??!?D?+??R      ??!       Z	?D?+???D?+??!?D?+??b      ??!       JGPUY?:'??M??b q???????y8o?X@