# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework.formats import classification_pb2 as mediapipe_dot_framework_dot_formats_dot_classification__pb2
from mediapipe.framework.formats import landmark_pb2 as mediapipe_dot_framework_dot_formats_dot_landmark__pb2
from mediapipe.framework.formats import rect_pb2 as mediapipe_dot_framework_dot_formats_dot_rect__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.proto',
  package='mediapipe.tasks.containers.proto',
  syntax='proto2',
  serialized_options=b'\n6com.google.mediapipe.tasks.components.containers.protoB\035LandmarksDetectionResultProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nOmediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.proto\x12 mediapipe.tasks.containers.proto\x1a\x30mediapipe/framework/formats/classification.proto\x1a*mediapipe/framework/formats/landmark.proto\x1a&mediapipe/framework/formats/rect.proto\"\xe3\x01\n\x18LandmarksDetectionResult\x12\x34\n\tlandmarks\x18\x01 \x01(\x0b\x32!.mediapipe.NormalizedLandmarkList\x12\x36\n\x0f\x63lassifications\x18\x02 \x01(\x0b\x32\x1d.mediapipe.ClassificationList\x12\x30\n\x0fworld_landmarks\x18\x03 \x01(\x0b\x32\x17.mediapipe.LandmarkList\x12\'\n\x04rect\x18\x04 \x01(\x0b\x32\x19.mediapipe.NormalizedRect\"\xe9\x01\n\x1dMultiLandmarksDetectionResult\x12\x34\n\tlandmarks\x18\x01 \x03(\x0b\x32!.mediapipe.NormalizedLandmarkList\x12\x36\n\x0f\x63lassifications\x18\x02 \x03(\x0b\x32\x1d.mediapipe.ClassificationList\x12\x30\n\x0fworld_landmarks\x18\x03 \x03(\x0b\x32\x17.mediapipe.LandmarkList\x12(\n\x05rects\x18\x04 \x03(\x0b\x32\x19.mediapipe.NormalizedRectBW\n6com.google.mediapipe.tasks.components.containers.protoB\x1dLandmarksDetectionResultProto'
  ,
  dependencies=[mediapipe_dot_framework_dot_formats_dot_classification__pb2.DESCRIPTOR,mediapipe_dot_framework_dot_formats_dot_landmark__pb2.DESCRIPTOR,mediapipe_dot_framework_dot_formats_dot_rect__pb2.DESCRIPTOR,])




_LANDMARKSDETECTIONRESULT = _descriptor.Descriptor(
  name='LandmarksDetectionResult',
  full_name='mediapipe.tasks.containers.proto.LandmarksDetectionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='landmarks', full_name='mediapipe.tasks.containers.proto.LandmarksDetectionResult.landmarks', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='classifications', full_name='mediapipe.tasks.containers.proto.LandmarksDetectionResult.classifications', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='world_landmarks', full_name='mediapipe.tasks.containers.proto.LandmarksDetectionResult.world_landmarks', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rect', full_name='mediapipe.tasks.containers.proto.LandmarksDetectionResult.rect', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=252,
  serialized_end=479,
)


_MULTILANDMARKSDETECTIONRESULT = _descriptor.Descriptor(
  name='MultiLandmarksDetectionResult',
  full_name='mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='landmarks', full_name='mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult.landmarks', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='classifications', full_name='mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult.classifications', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='world_landmarks', full_name='mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult.world_landmarks', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rects', full_name='mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult.rects', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=482,
  serialized_end=715,
)

_LANDMARKSDETECTIONRESULT.fields_by_name['landmarks'].message_type = mediapipe_dot_framework_dot_formats_dot_landmark__pb2._NORMALIZEDLANDMARKLIST
_LANDMARKSDETECTIONRESULT.fields_by_name['classifications'].message_type = mediapipe_dot_framework_dot_formats_dot_classification__pb2._CLASSIFICATIONLIST
_LANDMARKSDETECTIONRESULT.fields_by_name['world_landmarks'].message_type = mediapipe_dot_framework_dot_formats_dot_landmark__pb2._LANDMARKLIST
_LANDMARKSDETECTIONRESULT.fields_by_name['rect'].message_type = mediapipe_dot_framework_dot_formats_dot_rect__pb2._NORMALIZEDRECT
_MULTILANDMARKSDETECTIONRESULT.fields_by_name['landmarks'].message_type = mediapipe_dot_framework_dot_formats_dot_landmark__pb2._NORMALIZEDLANDMARKLIST
_MULTILANDMARKSDETECTIONRESULT.fields_by_name['classifications'].message_type = mediapipe_dot_framework_dot_formats_dot_classification__pb2._CLASSIFICATIONLIST
_MULTILANDMARKSDETECTIONRESULT.fields_by_name['world_landmarks'].message_type = mediapipe_dot_framework_dot_formats_dot_landmark__pb2._LANDMARKLIST
_MULTILANDMARKSDETECTIONRESULT.fields_by_name['rects'].message_type = mediapipe_dot_framework_dot_formats_dot_rect__pb2._NORMALIZEDRECT
DESCRIPTOR.message_types_by_name['LandmarksDetectionResult'] = _LANDMARKSDETECTIONRESULT
DESCRIPTOR.message_types_by_name['MultiLandmarksDetectionResult'] = _MULTILANDMARKSDETECTIONRESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LandmarksDetectionResult = _reflection.GeneratedProtocolMessageType('LandmarksDetectionResult', (_message.Message,), {
  'DESCRIPTOR' : _LANDMARKSDETECTIONRESULT,
  '__module__' : 'mediapipe.tasks.cc.components.containers.proto.landmarks_detection_result_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.tasks.containers.proto.LandmarksDetectionResult)
  })
_sym_db.RegisterMessage(LandmarksDetectionResult)

MultiLandmarksDetectionResult = _reflection.GeneratedProtocolMessageType('MultiLandmarksDetectionResult', (_message.Message,), {
  'DESCRIPTOR' : _MULTILANDMARKSDETECTIONRESULT,
  '__module__' : 'mediapipe.tasks.cc.components.containers.proto.landmarks_detection_result_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.tasks.containers.proto.MultiLandmarksDetectionResult)
  })
_sym_db.RegisterMessage(MultiLandmarksDetectionResult)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
