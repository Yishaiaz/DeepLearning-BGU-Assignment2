	??~??8k@??~??8k@!??~??8k@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??~??8k@??;?H`@1??@-mU@Aۤ???w??I???'??*	?(\??.?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??6??6@!}?????X@)?O?cr6@1??+?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@q???im??!?6????)9{g?UI??1?????S??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@??ek}???!nQ?5Yq??)??l#???1????,J??:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@????????!d+ˤXN??)????????1d+ˤXN??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetcho????'??!???X?b??)o????'??1???X?b??:Preprocessing2F
Iterator::Model?5??
??!??rp!??)|?q7??1z6??Y??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism2?CP5z??!??춽8??)???QI??1?K*???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???avRN@QC~???C@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??;?H`@??;?H`@!??;?H`@      ??!       "	??@-mU@??@-mU@!??@-mU@*      ??!       2	ۤ???w??ۤ???w??!ۤ???w??:	???'?????'??!???'??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???avRN@yC~???C@