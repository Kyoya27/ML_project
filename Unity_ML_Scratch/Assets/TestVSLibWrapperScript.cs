using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestVSLibWrapperScript : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
        Debug.Log(ClassVSWrapper.my_add(42, 51));
        Debug.Log(ClassVSWrapper.my_mul(2, 3));
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
