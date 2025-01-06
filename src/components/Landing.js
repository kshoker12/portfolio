import React from 'react'
import mainbg from '../images/mainbg.jpg'

/**
 * @description Main Landing page of the application highlighting important information
 * @returns {Element}
 */
const Landing = () => {

    /** Render background with 3d Model */
    return (
        <div 
            className='h-screen w-screen bg-center bg-cover'
            style={{ backgroundImage: `url(${mainbg})` }}
        >
            <div className='w-full h-full absolute opacity-20 bg-alt'/>
        </div>
    );
};

export default Landing;