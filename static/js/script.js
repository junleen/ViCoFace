let slideIndex = 0;

function slideRight() {
    const slides = document.querySelectorAll('.video-slide');
    slideIndex = (slideIndex + 1) % slides.length; // 使用模运算来循环
    updateSlidePosition();
}

function slideLeft() {
    const slides = document.querySelectorAll('.video-slide');
    slideIndex = (slideIndex - 1 + slides.length) % slides.length; // 使用模运算来循环
    updateSlidePosition();
}


function updateSlidePosition() {
    const videoContainer = document.querySelector('.video-container');
    videoContainer.style.transform = `translateX(${slideIndex * -100}%)`;
}
