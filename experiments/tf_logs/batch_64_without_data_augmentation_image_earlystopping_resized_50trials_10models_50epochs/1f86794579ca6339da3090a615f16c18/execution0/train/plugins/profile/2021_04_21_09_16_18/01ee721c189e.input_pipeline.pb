	yGsdTZ@yGsdTZ@!yGsdTZ@	???$?????$??!???$??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6yGsdTZ@?'Hlwg,@1?A??)V@A???^(`??I??? ???Y?E?~`??*	Zd;ߝ?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2I?L???@!??????X@)U??N?@1~??^)>W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@<??f???!??뜴?@)??@fgѷ?1??0ZYd@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??f׽?!???@).???쟯?1vD;??k??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@R??m???!T?????)R??m???1T?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???????!???N???)???????1???N???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??gϱ?!??1r???)zVҊo(??1ꠡ?F???:Preprocessing2F
Iterator::Model5bf??(??![??Ֆ??)?f???u?1N?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???$??I$?ܰ2.@QK9?_U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?'Hlwg,@?'Hlwg,@!?'Hlwg,@      ??!       "	?A??)V@?A??)V@!?A??)V@*      ??!       2	???^(`?????^(`??!???^(`??:	??? ?????? ???!??? ???B      ??!       J	?E?~`???E?~`??!?E?~`??R      ??!       Z	?E?~`???E?~`??!?E?~`??b      ??!       JGPUY???$??b q$?ܰ2.@yK9?_U@