/****************************************************************************
** Meta object code from reading C++ file 'RTIMULibDemo.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../RTIMULibDemo/RTIMULibDemo.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RTIMULibDemo.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_RTIMULibDemo[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x05,

 // slots: signature, parameters, type, tag, flags
      23,   13,   13,   13, 0x0a,
      49,   13,   13,   13, 0x0a,
      77,   13,   13,   13, 0x0a,
     104,   13,   13,   13, 0x0a,
     118,   13,   13,   13, 0x0a,
     136,   13,   13,   13, 0x0a,
     155,   13,   13,   13, 0x0a,
     176,   13,   13,   13, 0x0a,
     195,   13,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_RTIMULibDemo[] = {
    "RTIMULibDemo\0\0newIMU()\0onSelectFusionAlgorithm()\0"
    "onCalibrateAccelerometers()\0"
    "onCalibrateMagnetometers()\0onSelectIMU()\0"
    "onEnableGyro(int)\0onEnableAccel(int)\0"
    "onEnableCompass(int)\0onEnableDebug(int)\0"
    "newIMUData(RTIMU_DATA)\0"
};

void RTIMULibDemo::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        RTIMULibDemo *_t = static_cast<RTIMULibDemo *>(_o);
        switch (_id) {
        case 0: _t->newIMU(); break;
        case 1: _t->onSelectFusionAlgorithm(); break;
        case 2: _t->onCalibrateAccelerometers(); break;
        case 3: _t->onCalibrateMagnetometers(); break;
        case 4: _t->onSelectIMU(); break;
        case 5: _t->onEnableGyro((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->onEnableAccel((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->onEnableCompass((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->onEnableDebug((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->newIMUData((*reinterpret_cast< const RTIMU_DATA(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData RTIMULibDemo::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject RTIMULibDemo::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_RTIMULibDemo,
      qt_meta_data_RTIMULibDemo, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &RTIMULibDemo::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *RTIMULibDemo::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *RTIMULibDemo::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_RTIMULibDemo))
        return static_cast<void*>(const_cast< RTIMULibDemo*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int RTIMULibDemo::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void RTIMULibDemo::newIMU()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
