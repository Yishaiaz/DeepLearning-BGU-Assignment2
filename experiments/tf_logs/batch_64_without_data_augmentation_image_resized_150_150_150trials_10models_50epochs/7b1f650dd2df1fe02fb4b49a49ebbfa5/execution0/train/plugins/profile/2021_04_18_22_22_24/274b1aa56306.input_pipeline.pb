	??>??h@??>??h@!??>??h@	???/??????/???!???/???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??>??h@Sy=?,[@1?N?LU@A??ْU??I?Li?-A??Y? ??%s??*	/?$??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?s???*@!?6$(?X@)Ϟ??$X@1???}W@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???????!̊e??*@)$??S ???1oQ_??@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@? #????!)?k'Fm@)I,)w????13???c0??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@??????!?;q(???)??????1?;q(???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?S?<??!M?A??)?S?<??1M?A??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!?kI-?S??)m??]???1???????:Preprocessing2F
Iterator::Model??r??h??!?S?v?u??)???7?{v?1?GM"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???/???IY6???L@Q?K?g?E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Sy=?,[@Sy=?,[@!Sy=?,[@      ??!       "	?N?LU@?N?LU@!?N?LU@*      ??!       2	??ْU????ْU??!??ْU??:	?Li?-A???Li?-A??!?Li?-A??B      ??!       J	? ??%s??? ??%s??!? ??%s??R      ??!       Z	? ??%s??? ??%s??!? ??%s??b      ??!       JGPUY???/???b qY6???L@y?K?g?E@