	?U??N?k@?U??N?k@!?U??N?k@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?U??N?k@?l ]l?`@1d?&?$U@A???????I?#???4??*	?ZdS,?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2	?Vд?6@!????X@)?j???t6@1?m*ֹX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@V-?????!??L?Z??)?Ŧ?B??1Ii0q?N??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?J??????!???(	g??)?O?????1 4??L??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@:?ؗl<??!jx???):?ؗl<??1jx???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ȯb???!????????)?ȯb???1????????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismL??$wج?!}??????)p
+TT??1a߻??%??:Preprocessing2F
Iterator::Model8? ?س??!A??t??)^?/??v?1!Hqc*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?)??'?N@Q=??C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?l ]l?`@?l ]l?`@!?l ]l?`@      ??!       "	d?&?$U@d?&?$U@!d?&?$U@*      ??!       2	??????????????!???????:	?#???4???#???4??!?#???4??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?)??'?N@y=??C@