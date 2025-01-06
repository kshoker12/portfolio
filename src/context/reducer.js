/** Main Reducer for application */
const mainReducer = (state, action) => {
    switch (action.type) { 
        default: {
            action.type = "DONE";
            return {
                ...state
            };
        };
    };
};

export default mainReducer;