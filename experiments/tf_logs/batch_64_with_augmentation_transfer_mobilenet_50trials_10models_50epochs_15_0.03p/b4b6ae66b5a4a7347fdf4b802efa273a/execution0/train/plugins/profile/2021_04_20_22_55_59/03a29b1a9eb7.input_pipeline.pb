	?D???l@?D???l@!?D???l@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?D???l@;?v?/?a@1?hW!?#V@A???????I?-?l???*	??K7???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?%Z?$6@! ??	?X@)? O!?5@1???w?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@hY??????!ķ????)?oB!??1???????:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??~??λ?!҅6YD??)?D??]??1???&?f??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?????ѩ?!H|?z???)?????ѩ?1H|?z???:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@bhur????!?kC2???)bhur????1?kC2???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism%@7n??!~?j"???)_?D?
??1??G?[)??:Preprocessing2F
Iterator::ModelT?:???!) ^?J???)?????z?1?*4?m۝?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIs????N@Q?o?T;C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;?v?/?a@;?v?/?a@!;?v?/?a@      ??!       "	?hW!?#V@?hW!?#V@!?hW!?#V@*      ??!       2	??????????????!???????:	?-?l????-?l???!?-?l???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qs????N@y?o?T;C@