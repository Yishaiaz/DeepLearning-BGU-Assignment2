	:??˄f@:??˄f@!:??˄f@	J$?p????J$?p????!J$?p????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6:??˄f@ {????V@1Xr??aU@A???Q????I@??"2???Yݗ3????*	/????$?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2,-#??*@!6'?N?rX@)??u??%@1??R??T@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCache@?ZӼ???!R?)#?&@)b.????1?M??U #@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@??B????!?C??	?-@)???IӠ??1M??A??
@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::MemoryCacheImpl@???U?6??!.??j6??)???U?6??1.??j6??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???`???!?))b???)???`???1?))b???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???bc^??!K????? @)?WuV???1??#???:Preprocessing2F
Iterator::Model?$?9ϸ?!A{/v?@)
ףp=
w?1^??	h??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9I$?p????I?膒??I@Qt
???G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 {????V@ {????V@! {????V@      ??!       "	Xr??aU@Xr??aU@!Xr??aU@*      ??!       2	???Q???????Q????!???Q????:	@??"2???@??"2???!@??"2???B      ??!       J	ݗ3????ݗ3????!ݗ3????R      ??!       Z	ݗ3????ݗ3????!ݗ3????b      ??!       JGPUYI$?p????b q?膒??I@yt
???G@